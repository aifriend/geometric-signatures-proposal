"""Tests for the training loop."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from geometric_signatures.config import (
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    SeedConfig,
    TrainingConfig,
)
from geometric_signatures.motifs import MotifSwitches
from geometric_signatures.population import NeuralPopulationData
from geometric_signatures.training.trainer import (
    EpochMetrics,
    TrainResult,
    _variant_name,
    train_multi_seed,
    train_single_seed,
)


def _tiny_config(
    epochs: int = 2,
    hidden: int = 8,
    motifs: MotifSwitches | None = None,
) -> ExperimentConfig:
    """Create a minimal config for fast training tests."""
    if motifs is None:
        motifs = MotifSwitches(
            normalization_gain_modulation=True,
            attractor_dynamics=True,
            selective_gating=True,
            expansion_recoding=True,
        )
    return ExperimentConfig(
        experiment=ExperimentMeta(
            name="test_run",
            seed=42,
            run_group="test",
        ),
        tasks=("evidence_accumulation", "perceptual_discrimination"),
        motifs=motifs,
        training=TrainingConfig(
            optimizer="adam",
            lr=0.01,
            batch_size=4,
            epochs=epochs,
        ),
        model=ModelConfig(
            hidden_size=hidden,
            num_layers=1,
            cell_type="constrained_rnn",
            dale_law=False,
            sparse_connectivity=0.0,
            input_dim=2,
            output_dim=1,
        ),
    )


class TestTrainSingleSeed:
    """Tests for train_single_seed."""

    @pytest.mark.slow
    def test_returns_train_result(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=1)
        result = train_single_seed(config, seed=42, output_dir=tmp_path)
        assert isinstance(result, TrainResult)

    @pytest.mark.slow
    def test_metrics_per_epoch(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=3)
        result = train_single_seed(config, seed=42, output_dir=tmp_path)
        assert len(result.metrics) == 3
        for m in result.metrics:
            assert isinstance(m, EpochMetrics)
            assert isinstance(m.loss, float)
            assert isinstance(m.val_loss, float)

    @pytest.mark.slow
    def test_checkpoint_created(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=1)
        result = train_single_seed(config, seed=42, output_dir=tmp_path)
        assert Path(result.checkpoint_path).exists()

    @pytest.mark.slow
    def test_manifest_created(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=1)
        train_single_seed(config, seed=42, output_dir=tmp_path)
        manifest = tmp_path / "complete" / "seed_42" / "manifest.json"
        assert manifest.exists()

    @pytest.mark.slow
    def test_population_data_returned(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=1)
        result = train_single_seed(config, seed=42, output_dir=tmp_path)
        assert isinstance(result.population_data, NeuralPopulationData)
        assert result.population_data.source == "rnn"
        # 2 tasks × 4 batch_size = 8 trials
        assert result.population_data.n_trials == 8

    @pytest.mark.slow
    def test_task_losses_reported(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=1)
        result = train_single_seed(config, seed=42, output_dir=tmp_path)
        # At least one task should have a loss
        assert len(result.metrics[0].task_losses) > 0

    @pytest.mark.slow
    def test_with_catalog(self, tmp_path: Path) -> None:
        from geometric_signatures.tracking import ExperimentCatalog

        catalog = ExperimentCatalog(tmp_path / "catalog.db")
        config = _tiny_config(epochs=1)
        train_single_seed(config, seed=42, output_dir=tmp_path, catalog=catalog)
        records = catalog.query(status="completed")
        assert len(records) == 1
        assert records[0].seed == 42


class TestTrainMultiSeed:
    """Tests for train_multi_seed."""

    @pytest.mark.slow
    def test_returns_multiple_results(self, tmp_path: Path) -> None:
        config = ExperimentConfig(
            experiment=ExperimentMeta(name="test", seed=0, run_group="test"),
            tasks=("evidence_accumulation",),
            motifs=MotifSwitches(
                normalization_gain_modulation=True,
                attractor_dynamics=True,
                selective_gating=True,
                expansion_recoding=True,
            ),
            training=TrainingConfig(optimizer="adam", lr=0.01, batch_size=4, epochs=1),
            model=ModelConfig(
                hidden_size=8, num_layers=1, cell_type="constrained_rnn",
                dale_law=False, sparse_connectivity=0.0,
                input_dim=2, output_dim=1,
            ),
            seeds=SeedConfig(base_seed=10, n_seeds=2),
        )
        results = train_multi_seed(config, tmp_path)
        assert len(results) == 2
        assert results[0].seed == 10
        assert results[1].seed == 11


class TestProgressCallback:
    """Tests for progress_callback parameter."""

    @pytest.mark.slow
    def test_callback_called_each_epoch(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=3)
        received: list[EpochMetrics] = []
        train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=received.append,
        )
        assert len(received) == 3

    @pytest.mark.slow
    def test_callback_epoch_ordering(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=4)
        received: list[EpochMetrics] = []
        train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=received.append,
        )
        epochs = [m.epoch for m in received]
        assert epochs == [0, 1, 2, 3]

    @pytest.mark.slow
    def test_callback_matches_final_result(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=2)
        received: list[EpochMetrics] = []
        result = train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=received.append,
        )
        assert received[-1] == result.metrics[-1]

    @pytest.mark.slow
    def test_callback_none_backward_compat(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=1)
        result = train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=None,
        )
        assert isinstance(result, TrainResult)
        assert len(result.metrics) == 1

    @pytest.mark.slow
    def test_callback_exception_ignored(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=2)

        def bad_callback(m: EpochMetrics) -> None:
            raise RuntimeError("callback boom")

        result = train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=bad_callback,
        )
        # Training should complete despite callback errors
        assert len(result.metrics) == 2


class TestCancelEvent:
    """Tests for cancel_event parameter."""

    @pytest.mark.slow
    def test_cancel_stops_early(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=10)
        cancel = threading.Event()
        received: list[EpochMetrics] = []

        def callback_and_cancel(m: EpochMetrics) -> None:
            received.append(m)
            if m.epoch >= 2:
                cancel.set()

        result = train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=callback_and_cancel,
            cancel_event=cancel,
        )
        # Should have stopped well before 10 epochs
        assert len(result.metrics) < 10
        assert len(result.metrics) >= 3  # completed epochs 0, 1, 2 then cancel

    @pytest.mark.slow
    def test_cancel_returns_partial_result(self, tmp_path: Path) -> None:
        config = _tiny_config(epochs=10)
        cancel = threading.Event()

        def cancel_after_first(m: EpochMetrics) -> None:
            if m.epoch >= 1:
                cancel.set()

        result = train_single_seed(
            config, seed=42, output_dir=tmp_path,
            progress_callback=cancel_after_first,
            cancel_event=cancel,
        )
        assert isinstance(result, TrainResult)
        assert len(result.metrics) < 10
        # Checkpoint should still exist
        assert Path(result.checkpoint_path).exists()
        # Population data should still be valid
        assert isinstance(result.population_data, NeuralPopulationData)


class TestVariantName:
    """Tests for _variant_name."""

    def test_complete(self) -> None:
        motifs = MotifSwitches(
            normalization_gain_modulation=True,
            attractor_dynamics=True,
            selective_gating=True,
            expansion_recoding=True,
        )
        assert _variant_name(motifs) == "complete"

    def test_single_ablation(self) -> None:
        motifs = MotifSwitches(
            normalization_gain_modulation=True,
            attractor_dynamics=False,
            selective_gating=True,
            expansion_recoding=True,
        )
        assert _variant_name(motifs) == "ablate_attractor"

    def test_multi_ablation(self) -> None:
        motifs = MotifSwitches(
            normalization_gain_modulation=False,
            attractor_dynamics=False,
            selective_gating=True,
            expansion_recoding=True,
        )
        assert _variant_name(motifs) == "ablate_normalization_attractor"
