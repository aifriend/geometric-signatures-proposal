"""Core package for geometric motif experiments."""

from .config import (
    AnalysisConfig,
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    SeedConfig,
    TrainingConfig,
    load_experiment_config,
)
from .logging_config import setup_logging
from .motifs import MotifSwitches, build_single_ablation_variants
from .population import NeuralPopulationData, TrialMetadata
from .reproducibility import (
    capture_environment,
    enable_deterministic_mode,
    resolve_device,
    set_all_seeds,
)
from .tasks import REQUIRED_TASKS, validate_task_battery
from .tracking import ExperimentCatalog, RunRecord

__all__ = [
    # Config
    "ExperimentConfig",
    "ExperimentMeta",
    "TrainingConfig",
    "ModelConfig",
    "AnalysisConfig",
    "SeedConfig",
    "load_experiment_config",
    # Motifs
    "MotifSwitches",
    "build_single_ablation_variants",
    # Tasks
    "REQUIRED_TASKS",
    "validate_task_battery",
    # Population data
    "NeuralPopulationData",
    "TrialMetadata",
    # Reproducibility
    "set_all_seeds",
    "enable_deterministic_mode",
    "resolve_device",
    "capture_environment",
    # Logging
    "setup_logging",
    # Tracking
    "ExperimentCatalog",
    "RunRecord",
]
