"""Experiment configuration loading and validation.

Provides frozen dataclasses for the full experiment configuration schema:

- ``ExperimentMeta``: Experiment name, seed, run group.
- ``TrainingConfig``: Optimizer, learning rate, batch size, epochs.
- ``ModelConfig``: Constrained RNN architecture parameters.
- ``AnalysisConfig``: Analysis pipeline settings (methods, preprocessing, stats).
- ``SeedConfig``: Multi-seed execution parameters.
- ``ExperimentConfig``: Top-level config aggregating all sections.

Backward compatibility: Phase 1 YAML configs (without model/analysis/seeds)
still load correctly — new sections default to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .motifs import MotifSwitches
from .tasks import validate_task_battery


# ── Phase 1 dataclasses (unchanged) ───────────────────────────


@dataclass(frozen=True)
class ExperimentMeta:
    """Experiment metadata."""

    name: str
    seed: int
    run_group: str


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        optimizer: Optimizer name ("adam" or "sgd").
        lr: Learning rate.
        batch_size: Number of trials per training step.
        epochs: Total training epochs.
        steps_per_epoch: Gradient updates per epoch. Default 50 gives
            ``50 × batch_size`` samples per epoch (3,200 with batch_size=64).
        patience: Early-stopping patience in epochs.  ``0`` disables early
            stopping (default for backward compatibility).
        lr_scheduler: Learning-rate schedule: "cosine" (default), "plateau",
            or "none".
    """

    optimizer: str
    lr: float
    batch_size: int
    epochs: int
    steps_per_epoch: int = 50
    patience: int = 0
    lr_scheduler: str = "cosine"


# ── Phase 2 dataclasses (new) ─────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    """Constrained RNN model architecture.

    Attributes:
        hidden_size: Number of hidden units in the recurrent layer.
        num_layers: Number of stacked recurrent layers.
        cell_type: RNN cell type identifier (e.g., "constrained_rnn").
        dale_law: Whether to enforce Dale's law (excitatory/inhibitory separation).
        sparse_connectivity: Fraction of zero weights (0.0 = dense, 0.9 = 90% sparse).
        input_dim: Dimensionality of input signals.
        output_dim: Dimensionality of output (decision) signals.
    """

    hidden_size: int
    num_layers: int
    cell_type: str
    dale_law: bool
    sparse_connectivity: float
    input_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0.0 <= self.sparse_connectivity < 1.0:
            raise ValueError(
                f"sparse_connectivity must be in [0, 1), got {self.sparse_connectivity}"
            )


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis pipeline configuration.

    Attributes:
        methods: Analysis method names (validated against ANALYSIS_REGISTRY).
        n_components: PCA components for preprocessing (where applicable).
        max_homology_dim: Maximum homology dimension for persistent homology.
        n_permutations: Number of permutations for statistical testing.
        confidence_level: Confidence level for bootstrap CIs (e.g., 0.95).
        correction_method: Multiple comparison correction ("fdr_bh", "bonferroni", "holm").
    """

    methods: tuple[str, ...]
    n_components: int = 50
    max_homology_dim: int = 2
    n_permutations: int = 1000
    confidence_level: float = 0.95
    correction_method: str = "fdr_bh"

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if self.max_homology_dim < 0:
            raise ValueError(
                f"max_homology_dim must be >= 0, got {self.max_homology_dim}"
            )
        if self.n_permutations < 1:
            raise ValueError(
                f"n_permutations must be >= 1, got {self.n_permutations}"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        valid_corrections = ("fdr_bh", "bonferroni", "holm")
        if self.correction_method not in valid_corrections:
            raise ValueError(
                f"correction_method must be one of {valid_corrections}, "
                f"got '{self.correction_method}'"
            )


@dataclass(frozen=True)
class SeedConfig:
    """Multi-seed execution configuration.

    Attributes:
        base_seed: Starting seed value.
        n_seeds: Number of seeds per variant (default 10 per comp neuro convention).
    """

    base_seed: int
    n_seeds: int = 10

    def __post_init__(self) -> None:
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")

    def seed_list(self) -> tuple[int, ...]:
        """Generate the list of seeds: base_seed, base_seed+1, ..., base_seed+n_seeds-1."""
        return tuple(range(self.base_seed, self.base_seed + self.n_seeds))


# ── Top-level config (backward-compatible) ─────────────────────


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Phase 1 fields (experiment, tasks, motifs, training) are always required.
    Phase 2+ fields (model, analysis, seeds) are optional and default to None
    for backward compatibility with Phase 1 YAML configs.
    """

    experiment: ExperimentMeta
    tasks: tuple[str, ...]
    motifs: MotifSwitches
    training: TrainingConfig
    model: ModelConfig | None = None
    analysis: AnalysisConfig | None = None
    seeds: SeedConfig | None = None


# ── Config loading ─────────────────────────────────────────────


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _parse_model_config(raw: dict[str, Any]) -> ModelConfig | None:
    """Parse optional model section from YAML."""
    model_raw = raw.get("model")
    if model_raw is None:
        return None
    if not isinstance(model_raw, dict):
        raise ValueError("model section must be a mapping")
    return ModelConfig(
        hidden_size=int(_require(model_raw, "hidden_size")),
        num_layers=int(_require(model_raw, "num_layers")),
        cell_type=str(_require(model_raw, "cell_type")),
        dale_law=bool(_require(model_raw, "dale_law")),
        sparse_connectivity=float(_require(model_raw, "sparse_connectivity")),
        input_dim=int(_require(model_raw, "input_dim")),
        output_dim=int(_require(model_raw, "output_dim")),
    )


def _parse_analysis_config(raw: dict[str, Any]) -> AnalysisConfig | None:
    """Parse optional analysis section from YAML."""
    analysis_raw = raw.get("analysis")
    if analysis_raw is None:
        return None
    if not isinstance(analysis_raw, dict):
        raise ValueError("analysis section must be a mapping")
    methods = tuple(_require(analysis_raw, "methods"))
    kwargs: dict[str, Any] = {"methods": methods}
    # Optional fields with defaults from the dataclass
    for key in ("n_components", "max_homology_dim", "n_permutations"):
        if key in analysis_raw:
            kwargs[key] = int(analysis_raw[key])
    if "confidence_level" in analysis_raw:
        kwargs["confidence_level"] = float(analysis_raw["confidence_level"])
    if "correction_method" in analysis_raw:
        kwargs["correction_method"] = str(analysis_raw["correction_method"])
    return AnalysisConfig(**kwargs)


def _parse_seed_config(raw: dict[str, Any]) -> SeedConfig | None:
    """Parse optional seeds section from YAML."""
    seeds_raw = raw.get("seeds")
    if seeds_raw is None:
        return None
    if not isinstance(seeds_raw, dict):
        raise ValueError("seeds section must be a mapping")
    base_seed = int(_require(seeds_raw, "base_seed"))
    n_seeds = int(seeds_raw.get("n_seeds", 10))
    return SeedConfig(base_seed=base_seed, n_seeds=n_seeds)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Phase 1 YAML files (without model/analysis/seeds sections) are fully
    supported — new sections default to None.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        ValueError: On missing required keys or invalid values.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    exp = _require(raw, "experiment")
    task_battery = _require(raw, "task_battery")
    motifs = _require(raw, "motifs")
    training = _require(raw, "training")

    if not isinstance(exp, dict) or not isinstance(task_battery, dict):
        raise ValueError("experiment and task_battery must be mappings")

    tasks = validate_task_battery(tuple(_require(task_battery, "tasks")))

    return ExperimentConfig(
        experiment=ExperimentMeta(
            name=str(_require(exp, "name")),
            seed=int(_require(exp, "seed")),
            run_group=str(_require(exp, "run_group")),
        ),
        tasks=tasks,
        motifs=MotifSwitches.from_mapping(motifs),
        training=TrainingConfig(
            optimizer=str(_require(training, "optimizer")),
            lr=float(_require(training, "lr")),
            batch_size=int(_require(training, "batch_size")),
            epochs=int(_require(training, "epochs")),
            steps_per_epoch=int(training.get("steps_per_epoch", 50)),
            patience=int(training.get("patience", 0)),
            lr_scheduler=str(training.get("lr_scheduler", "cosine")),
        ),
        model=_parse_model_config(raw),
        analysis=_parse_analysis_config(raw),
        seeds=_parse_seed_config(raw),
    )
