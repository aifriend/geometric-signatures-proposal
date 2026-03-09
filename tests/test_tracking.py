"""Tests for experiment tracking: hashing, manifests, and SQLite catalog.

Phase 1 tests (hash, manifest, payload) are preserved unchanged.
Phase 2 adds tests for RunRecord validation and ExperimentCatalog CRUD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geometric_signatures.config import ExperimentConfig, ExperimentMeta, TrainingConfig
from geometric_signatures.motifs import MotifSwitches
from geometric_signatures.tracking import (
    ExperimentCatalog,
    RunRecord,
    dataclass_payload,
    stable_config_hash,
    write_run_manifest,
)


def _sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentMeta(name="test", seed=42, run_group="unit"),
        tasks=("context_dependent_integration", "evidence_accumulation",
               "working_memory", "perceptual_discrimination"),
        motifs=MotifSwitches(True, True, True, True),
        training=TrainingConfig(optimizer="adam", lr=0.001, batch_size=32, epochs=10),
    )


def _sample_record(
    variant: str = "complete",
    seed: int = 42,
    status: str = "completed",
) -> RunRecord:
    return RunRecord(
        config_hash="abc123def456",
        variant_name=variant,
        seed=seed,
        timestamp="2026-03-04T12:00:00",
        manifest_path="/runs/complete/seed_42/manifest.json",
        status=status,
        environment={"python_version": "3.10.12", "numpy_version": "1.24.0"},
    )


# ── Phase 1 tests (unchanged) ─────────────────────────────────


def test_stable_config_hash_is_deterministic() -> None:
    payload = {"seed": 42, "name": "test", "z_key": "last"}
    h1 = stable_config_hash(payload)
    h2 = stable_config_hash(payload)
    assert h1 == h2


def test_stable_config_hash_ignores_key_order() -> None:
    a = stable_config_hash({"b": 2, "a": 1})
    b = stable_config_hash({"a": 1, "b": 2})
    assert a == b


def test_stable_config_hash_differs_on_different_payload() -> None:
    h1 = stable_config_hash({"seed": 42})
    h2 = stable_config_hash({"seed": 43})
    assert h1 != h2


def test_stable_config_hash_returns_hex_string() -> None:
    h = stable_config_hash({"key": "value"})
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex length
    int(h, 16)  # must be valid hex


def test_write_run_manifest_creates_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "runs" / "manifest.json"
    payload = {"config_hash": "abc123", "seed": 42}
    write_run_manifest(manifest_path, payload)

    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["config_hash"] == "abc123"
    assert data["seed"] == 42


def test_write_run_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    deep_path = tmp_path / "a" / "b" / "c" / "manifest.json"
    write_run_manifest(deep_path, {"ok": True})
    assert deep_path.exists()


def test_dataclass_payload_converts_config() -> None:
    cfg = _sample_config()
    payload = dataclass_payload(cfg)

    assert isinstance(payload, dict)
    assert payload["experiment"]["name"] == "test"
    assert payload["experiment"]["seed"] == 42
    assert payload["motifs"]["attractor_dynamics"] is True
    assert payload["training"]["optimizer"] == "adam"


def test_dataclass_payload_converts_motif_switches() -> None:
    motifs = MotifSwitches(True, False, True, False)
    payload = dataclass_payload(motifs)

    assert payload["normalization_gain_modulation"] is True
    assert payload["attractor_dynamics"] is False
    assert payload["selective_gating"] is True
    assert payload["expansion_recoding"] is False


# ── Phase 2: RunRecord tests ──────────────────────────────────


class TestRunRecord:
    def test_valid_record(self) -> None:
        record = _sample_record()
        assert record.config_hash == "abc123def456"
        assert record.status == "completed"

    def test_rejects_invalid_status(self) -> None:
        with pytest.raises(ValueError, match="status must be one of"):
            RunRecord(
                config_hash="abc",
                variant_name="complete",
                seed=0,
                timestamp="2026-01-01T00:00:00",
                manifest_path="/tmp/m.json",
                status="invalid_status",
                environment={},
            )

    def test_valid_statuses(self) -> None:
        for status in ("completed", "failed", "running"):
            record = _sample_record(status=status)
            assert record.status == status


# ── Phase 2: ExperimentCatalog tests ──────────────────────────


class TestExperimentCatalog:
    def test_creates_database_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "catalog.db"
        with ExperimentCatalog(db_path) as catalog:
            assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db_path = tmp_path / "deep" / "nested" / "catalog.db"
        with ExperimentCatalog(db_path) as catalog:
            assert db_path.exists()

    def test_register_and_query_single_run(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            record = _sample_record()
            catalog.register_run(record)

            results = catalog.query()
            assert len(results) == 1
            assert results[0].config_hash == record.config_hash
            assert results[0].variant_name == "complete"
            assert results[0].seed == 42
            assert results[0].status == "completed"

    def test_register_multiple_runs(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            for seed in range(5):
                catalog.register_run(_sample_record(seed=seed))

            results = catalog.query()
            assert len(results) == 5

    def test_query_by_variant(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(variant="complete", seed=0))
            catalog.register_run(_sample_record(variant="ablate_attractor_dynamics", seed=0))
            catalog.register_run(_sample_record(variant="complete", seed=1))

            results = catalog.query(variant="complete")
            assert len(results) == 2
            assert all(r.variant_name == "complete" for r in results)

    def test_query_by_status(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(seed=0, status="completed"))
            catalog.register_run(_sample_record(seed=1, status="failed"))
            catalog.register_run(_sample_record(seed=2, status="running"))

            completed = catalog.query(status="completed")
            assert len(completed) == 1
            assert completed[0].seed == 0

    def test_query_by_seed(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(seed=42))
            catalog.register_run(_sample_record(seed=99))

            results = catalog.query(seed=42)
            assert len(results) == 1
            assert results[0].seed == 42

    def test_query_empty_result(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            results = catalog.query(variant="nonexistent")
            assert results == []

    def test_get_by_hash(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(seed=0))
            catalog.register_run(_sample_record(seed=1))

            results = catalog.get_by_hash("abc123def456")
            assert len(results) == 2

    def test_get_seeds_for_variant(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            for seed in [10, 5, 20, 15]:
                catalog.register_run(_sample_record(variant="complete", seed=seed))

            seeds = catalog.get_seeds_for_variant("complete")
            assert seeds == [5, 10, 15, 20]  # sorted

    def test_count(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(variant="complete", seed=0))
            catalog.register_run(_sample_record(variant="complete", seed=1, status="failed"))
            catalog.register_run(_sample_record(variant="ablate_attractor_dynamics", seed=0))

            assert catalog.count() == 3
            assert catalog.count(variant="complete") == 2
            assert catalog.count(status="failed") == 1
            assert catalog.count(variant="complete", status="completed") == 1

    def test_update_status(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(seed=0, status="running"))

            # Verify running
            results = catalog.query(status="running")
            assert len(results) == 1

            # Update to completed
            catalog.update_status("abc123def456", "complete", 0, "completed")
            results = catalog.query(status="completed")
            assert len(results) == 1
            assert results[0].seed == 0

    def test_update_status_rejects_invalid(self, tmp_path: Path) -> None:
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            with pytest.raises(ValueError, match="status must be one of"):
                catalog.update_status("hash", "variant", 0, "bad_status")

    def test_register_updates_on_duplicate(self, tmp_path: Path) -> None:
        """INSERT OR REPLACE: same (hash, variant, seed) overwrites."""
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            catalog.register_run(_sample_record(seed=0, status="running"))
            catalog.register_run(_sample_record(seed=0, status="completed"))

            results = catalog.query()
            assert len(results) == 1
            assert results[0].status == "completed"

    def test_environment_round_trips(self, tmp_path: Path) -> None:
        """Environment dict is serialized to JSON and back."""
        with ExperimentCatalog(tmp_path / "catalog.db") as catalog:
            record = _sample_record()
            catalog.register_run(record)

            results = catalog.query()
            assert results[0].environment == record.environment
            assert results[0].environment["python_version"] == "3.10.12"

    def test_path_property(self, tmp_path: Path) -> None:
        db_path = tmp_path / "catalog.db"
        with ExperimentCatalog(db_path) as catalog:
            assert catalog.path == db_path

    def test_persistence_across_sessions(self, tmp_path: Path) -> None:
        """Data persists after closing and reopening the catalog."""
        db_path = tmp_path / "catalog.db"

        # Session 1: write
        with ExperimentCatalog(db_path) as catalog:
            catalog.register_run(_sample_record(seed=0))

        # Session 2: read
        with ExperimentCatalog(db_path) as catalog:
            results = catalog.query()
            assert len(results) == 1
            assert results[0].seed == 0
