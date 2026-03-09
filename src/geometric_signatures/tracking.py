"""Experiment tracking: hashing, manifests, and SQLite-backed run catalog.

Phase 1 utilities (``stable_config_hash``, ``write_run_manifest``,
``dataclass_payload``) are preserved unchanged.

Phase 2 adds ``RunRecord`` and ``ExperimentCatalog`` — a SQLite-backed
registry that scales to hundreds of runs (5 variants x 10 seeds x sensitivity
sweeps) with proper querying support.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


# ── Phase 1 utilities (unchanged) ──────────────────────────────


def stable_config_hash(config_payload: dict[str, Any]) -> str:
    """SHA-256 hash of a JSON-serialized dict (sorted keys, no whitespace).

    The same config always yields the same hash, regardless of Python dict
    ordering. Used to identify unique experiment configurations.
    """
    encoded = json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def write_run_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON manifest file. Creates parent directories if needed."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def dataclass_payload(value: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dict via ``dataclasses.asdict``."""
    return asdict(value)


# ── Phase 2: Run record & experiment catalog ───────────────────


@dataclass(frozen=True)
class RunRecord:
    """Metadata for a single training/analysis run.

    Attributes:
        config_hash: SHA-256 hash of the experiment config payload.
        variant_name: Ablation variant name (e.g., "complete", "ablate_attractor_dynamics").
        seed: Random seed used for this run.
        timestamp: ISO-format timestamp of run start.
        manifest_path: Path to the run's JSON manifest file.
        status: Run status — "completed", "failed", or "running".
        environment: Runtime environment dict (from ``capture_environment``).
    """

    config_hash: str
    variant_name: str
    seed: int
    timestamp: str
    manifest_path: str
    status: str
    environment: dict[str, str]

    def __post_init__(self) -> None:
        valid_statuses = ("completed", "failed", "running")
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got '{self.status}'"
            )


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('completed', 'failed', 'running')),
    environment TEXT NOT NULL,
    UNIQUE(config_hash, variant_name, seed)
)
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO runs
    (config_hash, variant_name, seed, timestamp, manifest_path, status, environment)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_STATUS_SQL = """
UPDATE runs SET status = ? WHERE config_hash = ? AND variant_name = ? AND seed = ?
"""


class ExperimentCatalog:
    """SQLite-backed experiment run registry.

    Provides persistent tracking of all training/analysis runs with query
    support by variant, seed, status, and config hash. Single-file storage
    that scales to hundreds of runs.

    Usage::

        catalog = ExperimentCatalog(Path("runs/catalog.db"))
        catalog.register_run(record)
        completed = catalog.query(status="completed")
        seeds = catalog.get_seeds_for_variant("complete")
    """

    def __init__(self, catalog_path: Path) -> None:
        """Initialize or open an existing catalog database.

        Args:
            catalog_path: Path to the SQLite database file. Parent directories
                are created if they don't exist.
        """
        self._path = catalog_path
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(catalog_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    @property
    def path(self) -> Path:
        """Path to the catalog database file."""
        return self._path

    def register_run(self, record: RunRecord) -> None:
        """Register a run in the catalog. Updates if (hash, variant, seed) exists.

        Args:
            record: RunRecord to register.
        """
        env_json = json.dumps(record.environment, sort_keys=True)
        self._conn.execute(
            _INSERT_SQL,
            (
                record.config_hash,
                record.variant_name,
                record.seed,
                record.timestamp,
                record.manifest_path,
                record.status,
                env_json,
            ),
        )
        self._conn.commit()

    def update_status(
        self, config_hash: str, variant_name: str, seed: int, status: str
    ) -> None:
        """Update the status of an existing run.

        Args:
            config_hash: Config hash identifying the run.
            variant_name: Variant name.
            seed: Seed value.
            status: New status ("completed", "failed", or "running").
        """
        valid_statuses = ("completed", "failed", "running")
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}, got '{status}'")
        self._conn.execute(
            _UPDATE_STATUS_SQL, (status, config_hash, variant_name, seed)
        )
        self._conn.commit()

    def query(
        self,
        variant: str | None = None,
        seed: int | None = None,
        status: str | None = None,
        config_hash: str | None = None,
    ) -> list[RunRecord]:
        """Query runs by optional filters.

        All filters are ANDed. Returns empty list if no matches.

        Args:
            variant: Filter by variant name.
            seed: Filter by seed value.
            status: Filter by run status.
            config_hash: Filter by config hash.

        Returns:
            List of matching RunRecord instances.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if variant is not None:
            conditions.append("variant_name = ?")
            params.append(variant)
        if seed is not None:
            conditions.append("seed = ?")
            params.append(seed)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if config_hash is not None:
            conditions.append("config_hash = ?")
            params.append(config_hash)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT config_hash, variant_name, seed, timestamp, manifest_path, status, environment FROM runs{where} ORDER BY timestamp"

        cursor = self._conn.execute(sql, params)
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_by_hash(self, config_hash: str) -> list[RunRecord]:
        """Get all runs matching a specific config hash.

        Args:
            config_hash: SHA-256 config hash.

        Returns:
            List of RunRecord instances.
        """
        return self.query(config_hash=config_hash)

    def get_seeds_for_variant(self, variant: str) -> list[int]:
        """Get all seed values used for a given variant.

        Args:
            variant: Variant name.

        Returns:
            Sorted list of seed values.
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT seed FROM runs WHERE variant_name = ? ORDER BY seed",
            (variant,),
        )
        return [row[0] for row in cursor.fetchall()]

    def count(
        self, variant: str | None = None, status: str | None = None
    ) -> int:
        """Count runs matching optional filters.

        Args:
            variant: Filter by variant name.
            status: Filter by run status.

        Returns:
            Number of matching runs.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if variant is not None:
            conditions.append("variant_name = ?")
            params.append(variant)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM runs{where}", params)
        return cursor.fetchone()[0]  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> ExperimentCatalog:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> RunRecord:
        """Convert a database row to a RunRecord."""
        return RunRecord(
            config_hash=row[0],
            variant_name=row[1],
            seed=row[2],
            timestamp=row[3],
            manifest_path=row[4],
            status=row[5],
            environment=json.loads(row[6]),
        )
