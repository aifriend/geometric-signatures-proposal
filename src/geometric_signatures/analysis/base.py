"""Analysis method protocol and result container.

Defines the ``AnalysisMethod`` :class:`~typing.Protocol` — a structural
subtype contract that any analysis method must satisfy. Using Protocol
rather than ABC means analysis methods need only have the right shape
(``name: str`` and ``compute(NeuralPopulationData) -> AnalysisResult``);
they don't need to inherit from anything.

Also defines ``AnalysisResult`` — a frozen dataclass for carrying analysis
outputs (arrays + scalars) with save/load persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnalysisMethod(Protocol):
    """Protocol for analysis methods.

    Any class with ``name: str`` and a ``compute`` method returning
    an ``AnalysisResult`` satisfies this protocol — no inheritance needed.
    """

    name: str

    def compute(self, data: Any) -> AnalysisResult:
        """Run the analysis on neural population data.

        Args:
            data: NeuralPopulationData instance.

        Returns:
            AnalysisResult with method-specific arrays and scalars.
        """
        ...


@dataclass(frozen=True)
class AnalysisResult:
    """Frozen container for analysis outputs.

    Attributes:
        method: Name of the analysis method that produced this result.
        config_hash: Deterministic hash of the experiment config.
        seed: Random seed used for the training run.
        variant: Ablation variant name.
        arrays: Named numpy arrays (embeddings, diagrams, RDMs, etc.).
        scalars: Named scalar values (distances, dimensions, etc.).
    """

    method: str
    config_hash: str
    seed: int
    variant: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save result to disk as .npz (arrays) + .json (metadata).

        Creates parent directories if needed. Files written:
        - ``<path>.npz``: numpy arrays
        - ``<path>.json``: scalars + metadata

        Args:
            path: Base path (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save arrays
        if self.arrays:
            np.savez(str(path) + ".npz", **self.arrays)  # type: ignore[arg-type]

        # Save metadata + scalars
        meta = {
            "method": self.method,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "variant": self.variant,
            "scalars": self.scalars,
            "array_keys": list(self.arrays.keys()),
        }
        json_path = Path(str(path) + ".json")
        json_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> AnalysisResult:
        """Load result from disk.

        Args:
            path: Base path (without extension). Expects ``.npz`` and ``.json``.

        Returns:
            Reconstructed AnalysisResult.

        Raises:
            FileNotFoundError: If the JSON metadata file doesn't exist.
        """
        path = Path(path)
        json_path = Path(str(path) + ".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        meta = json.loads(json_path.read_text())

        arrays: dict[str, np.ndarray] = {}
        npz_path = Path(str(path) + ".npz")
        if npz_path.exists():
            with np.load(str(npz_path)) as data:
                for key in data.files:
                    arrays[key] = data[key]

        return cls(
            method=meta["method"],
            config_hash=meta["config_hash"],
            seed=meta["seed"],
            variant=meta["variant"],
            arrays=arrays,
            scalars=meta.get("scalars", {}),
        )
