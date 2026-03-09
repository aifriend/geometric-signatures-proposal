"""Tests for experiment configuration loading and validation.

Validates:
- Phase 1 baseline YAML loads correctly.
- Phase 2 YAML with model/analysis/seeds sections loads correctly.
- Backward compatibility: Phase 1 YAML yields model=None, analysis=None, seeds=None.
- Negative tests: missing keys, invalid values, malformed YAML.
- New dataclass validation: ModelConfig, AnalysisConfig, SeedConfig.
"""

from pathlib import Path

import pytest

from geometric_signatures.config import (
    AnalysisConfig,
    ExperimentConfig,
    ModelConfig,
    SeedConfig,
    load_experiment_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG = PROJECT_ROOT / "config" / "experiment.baseline.yaml"
ABLATION_TEMPLATE = PROJECT_ROOT / "config" / "experiment.ablation_template.yaml"


# ── Phase 1 backward compatibility ────────────────────────────


def test_load_experiment_config_parses_baseline_file() -> None:
    cfg = load_experiment_config(BASELINE_CONFIG)

    assert cfg.experiment.name == "baseline_four_motif_complete"
    assert cfg.experiment.seed == 42
    assert cfg.motifs.attractor_dynamics is True
    assert len(cfg.tasks) == 4
    assert cfg.training.epochs == 50


def test_baseline_config_has_none_for_new_sections() -> None:
    """Phase 1 YAML → model=None, analysis=None, seeds=None."""
    cfg = load_experiment_config(BASELINE_CONFIG)
    assert cfg.model is None
    assert cfg.analysis is None
    assert cfg.seeds is None


# ── Phase 2 config with all sections ──────────────────────────


def test_load_ablation_template_parses_all_sections() -> None:
    cfg = load_experiment_config(ABLATION_TEMPLATE)

    # Phase 1 sections
    assert cfg.experiment.name == "ablation_template"
    assert len(cfg.tasks) == 4
    assert cfg.motifs.normalization_gain_modulation is True
    assert cfg.training.epochs == 200
    assert cfg.training.steps_per_epoch == 50
    assert cfg.training.patience == 0
    assert cfg.training.lr_scheduler == "cosine"

    # Phase 2 model section
    assert cfg.model is not None
    assert cfg.model.hidden_size == 256
    assert cfg.model.num_layers == 1
    assert cfg.model.cell_type == "constrained_rnn"
    assert cfg.model.dale_law is True
    assert cfg.model.sparse_connectivity == 0.1
    assert cfg.model.input_dim == 4
    assert cfg.model.output_dim == 1

    # Phase 2 analysis section
    assert cfg.analysis is not None
    assert "persistent_homology" in cfg.analysis.methods
    assert "rsa" in cfg.analysis.methods
    assert cfg.analysis.n_components == 50
    assert cfg.analysis.n_permutations == 1000
    assert cfg.analysis.correction_method == "fdr_bh"

    # Phase 2 seeds section
    assert cfg.seeds is not None
    assert cfg.seeds.base_seed == 0
    assert cfg.seeds.n_seeds == 10


# ── ModelConfig validation ─────────────────────────────────────


class TestModelConfig:
    def test_valid_model_config(self) -> None:
        mc = ModelConfig(
            hidden_size=128, num_layers=1, cell_type="constrained_rnn",
            dale_law=True, sparse_connectivity=0.1, input_dim=16, output_dim=4,
        )
        assert mc.hidden_size == 128

    def test_rejects_zero_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be >= 1"):
            ModelConfig(
                hidden_size=0, num_layers=1, cell_type="rnn",
                dale_law=False, sparse_connectivity=0.0, input_dim=1, output_dim=1,
            )

    def test_rejects_zero_num_layers(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            ModelConfig(
                hidden_size=32, num_layers=0, cell_type="rnn",
                dale_law=False, sparse_connectivity=0.0, input_dim=1, output_dim=1,
            )

    def test_rejects_invalid_sparsity(self) -> None:
        with pytest.raises(ValueError, match="sparse_connectivity must be in"):
            ModelConfig(
                hidden_size=32, num_layers=1, cell_type="rnn",
                dale_law=False, sparse_connectivity=1.0, input_dim=1, output_dim=1,
            )

    def test_rejects_negative_sparsity(self) -> None:
        with pytest.raises(ValueError, match="sparse_connectivity must be in"):
            ModelConfig(
                hidden_size=32, num_layers=1, cell_type="rnn",
                dale_law=False, sparse_connectivity=-0.1, input_dim=1, output_dim=1,
            )


# ── AnalysisConfig validation ──────────────────────────────────


class TestAnalysisConfig:
    def test_valid_analysis_config(self) -> None:
        ac = AnalysisConfig(methods=("rsa", "cka"))
        assert ac.n_components == 50  # default
        assert ac.n_permutations == 1000  # default
        assert ac.correction_method == "fdr_bh"  # default

    def test_rejects_invalid_n_components(self) -> None:
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            AnalysisConfig(methods=("rsa",), n_components=0)

    def test_rejects_invalid_n_permutations(self) -> None:
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            AnalysisConfig(methods=("rsa",), n_permutations=0)

    def test_rejects_invalid_confidence_level(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in"):
            AnalysisConfig(methods=("rsa",), confidence_level=1.0)

    def test_rejects_invalid_correction_method(self) -> None:
        with pytest.raises(ValueError, match="correction_method must be one of"):
            AnalysisConfig(methods=("rsa",), correction_method="invalid")


# ── SeedConfig validation ──────────────────────────────────────


class TestSeedConfig:
    def test_valid_seed_config(self) -> None:
        sc = SeedConfig(base_seed=42, n_seeds=5)
        assert sc.base_seed == 42
        assert sc.n_seeds == 5

    def test_default_n_seeds(self) -> None:
        sc = SeedConfig(base_seed=0)
        assert sc.n_seeds == 10

    def test_seed_list(self) -> None:
        sc = SeedConfig(base_seed=10, n_seeds=3)
        assert sc.seed_list() == (10, 11, 12)

    def test_rejects_zero_n_seeds(self) -> None:
        with pytest.raises(ValueError, match="n_seeds must be >= 1"):
            SeedConfig(base_seed=0, n_seeds=0)


# ── Phase 1 negative tests (unchanged) ────────────────────────


def test_load_config_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_experiment_config(empty)


def test_load_config_rejects_missing_experiment_key(tmp_path: Path) -> None:
    bad = tmp_path / "no_experiment.yaml"
    bad.write_text(
        "task_battery:\n  tasks: [context_dependent_integration,"
        " evidence_accumulation, working_memory, perceptual_discrimination]\n"
        "motifs:\n  normalization_gain_modulation: true\n"
        "  attractor_dynamics: true\n  selective_gating: true\n"
        "  expansion_recoding: true\n"
        "training:\n  optimizer: adam\n  lr: 0.001\n"
        "  batch_size: 32\n  epochs: 10\n"
    )
    with pytest.raises(ValueError, match="Missing required key: experiment"):
        load_experiment_config(bad)


def test_load_config_rejects_missing_motifs_key(tmp_path: Path) -> None:
    bad = tmp_path / "no_motifs.yaml"
    bad.write_text(
        "experiment:\n  name: test\n  seed: 1\n  run_group: test\n"
        "task_battery:\n  tasks: [context_dependent_integration,"
        " evidence_accumulation, working_memory, perceptual_discrimination]\n"
        "training:\n  optimizer: adam\n  lr: 0.001\n"
        "  batch_size: 32\n  epochs: 10\n"
    )
    with pytest.raises(ValueError, match="Missing required key: motifs"):
        load_experiment_config(bad)


def test_load_config_rejects_missing_training_key(tmp_path: Path) -> None:
    bad = tmp_path / "no_training.yaml"
    bad.write_text(
        "experiment:\n  name: test\n  seed: 1\n  run_group: test\n"
        "task_battery:\n  tasks: [context_dependent_integration,"
        " evidence_accumulation, working_memory, perceptual_discrimination]\n"
        "motifs:\n  normalization_gain_modulation: true\n"
        "  attractor_dynamics: true\n  selective_gating: true\n"
        "  expansion_recoding: true\n"
    )
    with pytest.raises(ValueError, match="Missing required key: training"):
        load_experiment_config(bad)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    bad = tmp_path / "list_root.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_experiment_config(bad)


def test_load_config_rejects_missing_motif_field(tmp_path: Path) -> None:
    bad = tmp_path / "partial_motifs.yaml"
    bad.write_text(
        "experiment:\n  name: test\n  seed: 1\n  run_group: test\n"
        "task_battery:\n  tasks: [context_dependent_integration,"
        " evidence_accumulation, working_memory, perceptual_discrimination]\n"
        "motifs:\n  normalization_gain_modulation: true\n"
        "  attractor_dynamics: true\n"
        "training:\n  optimizer: adam\n  lr: 0.001\n"
        "  batch_size: 32\n  epochs: 10\n"
    )
    with pytest.raises(ValueError, match="Missing motif keys"):
        load_experiment_config(bad)
